#!/usr/bin/env python3
"""Enforce platform-cfg encapsulation.

The rule (CLAUDE.md, "Code Style" -> "Platform cfg is encapsulation, not
sprinkle"): a *platform-predicate* `#[cfg(...)]` (target_os, target_arch,
target_family, target_feature, target_pointer_width, target_endian, windows,
unix, or a bare arch/os name like aarch64/x86_64/wasm32/macos/linux) may only
appear in one of three shapes:

  1. `#![cfg(...)]` -- an inner attribute. Rust only accepts these at the top
     of a file or block, so by construction they always gate a *whole* file
     or module, never a fragment of one.
  2. `#[cfg(...)]` directly above a `mod foo;` or `mod foo { ... }` item --
     gates a whole file (or whole inline submodule), wherever that mod item
     lives.
  3. `#[cfg(...)]` directly above a single-line item (ends in `;`, no `{`) --
     but only inside a file named `mod.rs`. This is the dispatch/re-export
     idiom (`#[cfg(aarch64)] pub use native::Foo as PlatformFoo;`) that
     selects a platform implementation without leaking the predicate into
     the implementation itself.

Anything else -- a platform cfg on a fn, struct, impl, field, const, enum
variant, or a multi-line item, living anywhere other than those two shapes --
means platform-specific behavior is threaded through a file that is supposed
to be platform-agnostic, rather than being encapsulated in its own file (or
selected once, at the seam, in mod.rs). That is exactly the shape this check
exists to catch: not a style nit, but the difference between "the platform
split is a file boundary" and "the platform split is grep -r cfg".

A line may opt out with a trailing `// cfg-encapsulation: <reason>` comment,
for a documented, reviewed exception.

This check is baselined: `cfg_encapsulation_baseline.txt` (next to this
script) lists every violation that predates the rule. Those are grandfathered
-- they still print, but don't fail the build -- while anything NOT in the
baseline fails it. That is deliberate: it turns the rule into a backstop
against new violations today, without demanding an audit-and-fix of ~200
files as the price of adding it. Run `--update-baseline` to accept the
current violation set (e.g. after fixing some, or knowingly adding one with
justification already reviewed in the PR) as the new baseline; don't run it
to silence a violation you haven't actually looked at.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "cfg_encapsulation_baseline.txt"

PLATFORM_PREDICATE = re.compile(
    r"\b("
    r"target_os|target_arch|target_family|target_feature|"
    r"target_pointer_width|target_endian|"
    r"windows|unix|macos|linux|aarch64|x86_64|wasm32"
    r")\b"
)

CFG_ATTR = re.compile(r"^\s*#\[\s*cfg\((.*)\)\s*\]\s*(.*)$")
CFG_INNER_ATTR = re.compile(r"^\s*#!\[\s*cfg\((.*)\)\s*\]")
MOD_ITEM = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*[;{]")
OTHER_ATTR_OR_COMMENT = re.compile(r"^\s*(#\[.*\]|//.*|/\*.*\*/)?\s*$")
OPT_OUT = re.compile(r"//\s*cfg-encapsulation:")

EXCLUDE_DIRS = {"target", "target.noindex", ".git"}


def iter_rust_files():
    for path in REPO_ROOT.rglob("*.rs"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def next_item_line(lines, idx):
    """First line after `idx` that isn't blank, a comment, or another attribute."""
    i = idx + 1
    while i < len(lines) and OTHER_ATTR_OR_COMMENT.match(lines[i]):
        i += 1
    return lines[i] if i < len(lines) else None


def check_file(path):
    violations = []
    is_mod_rs = path.name == "mod.rs"
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations
    lines = text.splitlines()

    for idx, line in enumerate(lines):
        if OPT_OUT.search(line):
            continue

        inner = CFG_INNER_ATTR.match(line)
        if inner:
            # Inner attributes are syntactically whole-file/whole-block --
            # always compliant regardless of predicate.
            continue

        m = CFG_ATTR.match(line)
        if not m:
            continue
        predicate, trailing = m.group(1), m.group(2).strip()
        if not PLATFORM_PREDICATE.search(predicate):
            continue

        # `#[cfg(unix)] mod foo;` puts the item on the same line as the
        # attribute -- a real Rust shape, not just the "attribute on its own
        # line, item below" one this checker started out only recognizing.
        if trailing and not trailing.startswith("//"):
            item = trailing
        else:
            item = next_item_line(lines, idx)
        if item is None:
            violations.append((path, idx + 1, line.strip(), "cfg at end of file, no item follows"))
            continue

        if MOD_ITEM.match(item):
            continue  # whole-file/whole-module gate, allowed anywhere

        stripped = item.strip()
        if is_mod_rs and stripped.endswith(";") and "{" not in stripped:
            continue  # single-line dispatch item, allowed in mod.rs

        violations.append((path, idx + 1, line.strip(), stripped))

    return violations


def baseline_key(path, cfg_line, item):
    """Identity used for baseline matching -- NOT line number, which shifts
    with unrelated edits to the file and would make every violation "new"
    the moment a line above it changes."""
    return f"{path.relative_to(REPO_ROOT)}\t{cfg_line}\t{item}"


def load_baseline():
    if not BASELINE_PATH.exists():
        return set()
    return {
        line
        for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def write_baseline(keys):
    header = (
        "# Grandfathered platform-cfg-encapsulation violations -- see\n"
        "# check_cfg_encapsulation.py's module docstring. Regenerate with\n"
        "# `python3 scripts/check_cfg_encapsulation.py --update-baseline`.\n"
        "# Format: <path>\\t<cfg line>\\t<item line>\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(sorted(keys)) + "\n", encoding="utf-8")


def main():
    argv = sys.argv[1:]
    update = "--update-baseline" in argv

    all_violations = []
    for path in sorted(iter_rust_files()):
        all_violations.extend(check_file(path))

    if update:
        write_baseline(baseline_key(p, c, i) for p, _, c, i in all_violations)
        print(f"Wrote {len(all_violations)} violation(s) to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    if not all_violations:
        print("OK: every platform #[cfg(...)] gates a whole file/mod, or is a single-line dispatch item in mod.rs")
        return 0

    baseline = load_baseline()
    rel = lambda p: p.relative_to(REPO_ROOT)
    new_violations = []
    grandfathered = 0

    for path, lineno, cfg_line, item in all_violations:
        key = baseline_key(path, cfg_line, item)
        if key in baseline:
            grandfathered += 1
            continue
        new_violations.append((path, lineno, cfg_line, item))
        print(
            f"::error file={rel(path)},line={lineno}::platform cfg does not gate a whole "
            f"file/mod, and this is not a single-line dispatch item in mod.rs: "
            f"`{cfg_line}` above `{item}`. Move the platform-specific code into its own "
            f"file gated by `#[cfg(...)] mod ...;` (or `#![cfg(...)]` at the top of that "
            f"file), and select it with a single-line item in mod.rs -- or add "
            f"`// cfg-encapsulation: <reason>` on this line for a reviewed exception.",
            file=sys.stderr,
        )

    if grandfathered:
        print(f"({grandfathered} pre-existing violation(s) grandfathered by {BASELINE_PATH.name} -- not fixed, not failing)")

    if not new_violations:
        print(f"OK: no new platform cfg encapsulation violations ({grandfathered} pre-existing, grandfathered)")
        return 0

    print(f"\nFAIL: {len(new_violations)} new platform cfg encapsulation violation(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
