#!/usr/bin/env python3
"""Enforce the emit boundary of pixelflow-codegen.

docs/plans/2026-09-12-emit-should-just-emit.md: the emitter -- `emit/`, the
register allocator included -- is handed a finished program and turns it into
bytes. It names nothing upstream of it: no optimizer, no lowering, no scoping,
no driver. And `program/`, the vocabulary and the stages that build it, names
nothing downstream: no allocator, no register, no byte.

The rules, over the non-test, non-comment, non-string text of each file:

  Rule E (every `pixelflow-codegen/src/emit/**/*.rs`) may not name
    - `pixelflow_search` (the optimizer),
    - `pixelflow_ir::passes`, `::variance` or `::store` (the passes, and the
      analysis lowering and scoping are made of),
    - a submodule of `program/` (`program::lower`, `program::scopes`,
      `program::guards`, ...): the emitter reads the vocabulary
      `program/mod.rs` defines (`IfGuard`, `ScopedSchedule`, `Def`, ...) and
      nothing behind it. The list is the filesystem's, not a hand-kept one, so
      a submodule added later is covered the day it is added,
    - `crate::pipeline`, `crate::jit_cache` (the driver and what drives it).
      The one exemption is a plain `pub use` re-export
      (`pub use crate::pipeline::compile;`): it keeps a
      public path where it was and is not a use. `pub(crate) use` and `use`
      are not re-exports and are not exempt, and nothing else is exempt from
      anything else.

  Rule P (every `pixelflow-codegen/src/program/**/*.rs`) may not name
    `crate::emit`, `crate::isa`, `crate::pipeline`, `crate::jit_cache`,
    `regalloc`, `executable` or `CompiledKernel`.
    `program/` MAY name `pixelflow_search`: pricing an arm is its business.

What is not scanned, deliberately: comments and string literals (prose and
panic messages name the stages freely), and any item gated by `#[cfg(test)]`
or `#[cfg(all(test, ..))]` (a test is a consumer, and reaches wherever it
likes). `#[cfg(not(test))]` and `#[cfg(any(test, ..))]` are NOT exempt: the
first is production code, the second is production code too.

Known loopholes, which a text scan cannot see: crate-root re-exports such as
`pixelflow_ir::LatticeShape`; `use ... as alias` renames, `use` globs and a
module imported then named by its short path (`use crate::program as p;
p::lower::x()`); macro-generated paths; and a method call into the driver from
non-test emit code. A file that is test-only by its parent's `mod`
declaration (`emit/coverage.rs`) is scanned as production, and so is a file
whose own `#![cfg(test)]` is an inner attribute (only `#[cfg(test)]` is
recognised) -- a false positive there is loud, never silent. An unrecognised
shape is always scanned, never skipped: the failure direction is more text,
not less.

There is no baseline file. Zero hits is the bar, and an exemption is a
reviewed edit to this script.

Usage:
  check_emit_boundary.py             scan the tree
  check_emit_boundary.py --self-test run the scanner over in-memory cases
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "pixelflow-codegen" / "src"

ITEM_KEYWORDS = (
    "fn", "mod", "impl", "struct", "enum", "trait", "type", "use", "const",
    "static", "extern", "union", "macro_rules",
)
BLOCK_KEYWORDS = ("if", "match", "for", "loop", "while")
CFG_TEST = re.compile(
    r"#\s*\[\s*cfg\s*\(\s*(test|all\s*\(\s*(?:[^()]*,\s*)?test\b[^)]*\))\s*\)\s*\]"
)
# A re-export is `pub use` exactly: `pub(crate) use` is a use with a narrower
# audience, not a public path kept in place.
PUB_USE = re.compile(r"\bpub\s+use\b[^;]*;")


def blank(text, a, b):
    """`text[a:b]` as spaces, newlines kept so line numbers stay true."""
    return text[:a] + "".join("\n" if c == "\n" else " " for c in text[a:b]) + text[b:]


def strip_comments_and_strings(src):
    """Comments, string literals (plain, raw, byte) and char literals, blanked."""
    out = list(src)
    i, n = 0, len(src)

    def blank_range(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank_range(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank_range(i, j)
            i = j
        elif (
            c in "rb"
            and re.match(r'(?:br|b|r)#*"', src[i:i + 12])
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            m = re.match(r'(b?)(r?)(#*)"', src[i:])
            prefix_len = len(m.group(1)) + len(m.group(2)) + len(m.group(3)) + 1
            raw = m.group(2) == "r"
            hashes = m.group(3)
            j = i + prefix_len
            if raw:
                end = src.find('"' + hashes, j)
                end = n if end < 0 else end + 1 + len(hashes)
            else:
                end = j
                while end < n and src[end] != '"':
                    end += 2 if src[end] == "\\" else 1
                end += 1
            blank_range(i, end)
            i = end
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank_range(i, j + 1)
            i = j + 1
        elif c == "'":
            if i + 1 < n and src[i + 1] == "\\":
                j = src.find("'", i + 3)
                blank_range(i, j + 1)
                i = j + 1
            elif i + 2 < n and src[i + 2] == "'":
                blank_range(i, i + 3)
                i += 3
            else:
                i += 1  # a lifetime, not a char literal
        else:
            i += 1
    return "".join(out)


def skip_ws(t, i):
    while i < len(t) and t[i].isspace():
        i += 1
    return i


def skip_attrs(t, i):
    """Index of the first token after any `#[..]` attributes at `i`."""
    while True:
        i = skip_ws(t, i)
        if t.startswith("#", i) and t[skip_ws(t, i + 1):].startswith("["):
            j = skip_ws(t, i + 1)
            depth = 0
            while j < len(t):
                if t[j] == "[":
                    depth += 1
                elif t[j] == "]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            i = j
        else:
            return i


def item_extent(t, i):
    """End of the item or fragment that starts at `i`.

    An item ends at its first top-level `;`, or at the `}` matching its first
    top-level `{`, whichever comes first. A fragment (a field, a struct-literal
    field, a statement, an expression) ends at its first depth-0 `,` or `;`, or
    at the matching `}` if it begins with one. EVERY extent also ends at an
    unmatched closer, not consuming it: without that a `#[cfg(test)]` last
    field with no trailing comma never meets a terminator and the blanking runs
    past the enclosing `}` into production code, which hides violations.
    """
    m = re.match(r"(?:pub\s*(?:\([^)]*\))?\s*)?(?:(?:unsafe|async|const)\s+)*", t[i:])
    k = i + (m.end() if m else 0)
    w = re.match(r"[A-Za-z_]+", t[k:])
    word = w.group(0) if w else ""
    is_item = word in ITEM_KEYWORDS
    depth = 0
    p = i
    n = len(t)
    if is_item:
        while p < n:
            c = t[p]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
                if depth < 0:
                    return p
            elif c == "{" and depth == 0:
                d = 1
                p += 1
                while p < n and d:
                    d += (t[p] == "{") - (t[p] == "}")
                    p += 1
                return p
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth < 0:
                    return p
            elif c == ";" and depth == 0:
                return p + 1
            p += 1
        return n
    if t.startswith("{", i):
        d = 0
        while p < n:
            d += (t[p] == "{") - (t[p] == "}")
            p += 1
            if d == 0:
                return p
        return n
    if word in BLOCK_KEYWORDS or (word == "" and t.startswith("{", k)):
        # A block-like statement (`if`, `match`, `for`, `loop`, `while`,
        # `unsafe { .. }`) ends at the `}` of its first block, not at the next
        # `,` or `;`, which belongs to the statement after it. An `else` chain
        # past that `}` stays scanned: more text, never less.
        while p < n:
            c = t[p]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
                if depth < 0:
                    return p
            elif c == "{" and depth == 0:
                d = 1
                p += 1
                while p < n and d:
                    d += (t[p] == "{") - (t[p] == "}")
                    p += 1
                return p
            elif c in ",;" and depth == 0:
                return p + 1
            p += 1
        return n
    while p < n:
        c = t[p]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return p
        elif c in ",;" and depth == 0:
            return p + 1
        p += 1
    return n


def blank_cfg_test(t):
    """Every item gated by `#[cfg(test)]` or `#[cfg(all(test, ..))]`, blanked."""
    pos = 0
    while True:
        m = CFG_TEST.search(t, pos)
        if not m:
            return t
        a = m.start()
        i = skip_attrs(t, m.end())
        end = item_extent(t, i)
        t = blank(t, a, end)
        pos = end


def cleaned(src):
    return blank_cfg_test(strip_comments_and_strings(src))


def blank_reexports(t):
    """Every plain `pub use ..;` statement, blanked."""
    return PUB_USE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), t)


def lineno(t, pos):
    return t.count("\n", 0, pos) + 1


def rules(program_submodules):
    """`{scope: [(pattern, exempt_reexports)]}`."""
    sub = "|".join(sorted(program_submodules))
    emit = [
        (r"\bpixelflow_search\b", False),
        (r"\bpixelflow_ir\s*::\s*(?:passes|variance|store)\b", False),
        (r"\bpixelflow_ir\s*::\s*\{[^;]*?\b(?:passes|variance|store)\b", False),
        (rf"\bprogram\s*::\s*(?:{sub})\b", False),
        (rf"\bprogram\s*::\s*\{{[^;]*?\b(?:{sub})\b", False),
        (r"\bprogram\s*::\s*\*", False),
        (r"\b(?:crate|super)\s*::\s*jit_cache\b", False),
        (r"\b(?:crate|super)\s*::\s*\{[^;]*?\bjit_cache\b", False),
        (r"\b(?:crate|super)\s*::\s*pipeline\b", True),
        (r"\b(?:crate|super)\s*::\s*\{[^;]*?\bpipeline\b", True),
    ]
    program = [
        (r"\b(?:crate|super)\s*::\s*(?:emit|isa|pipeline|jit_cache)\b", False),
        (r"\b(?:crate|super)\s*::\s*\{[^;]*?\b(?:emit|isa|pipeline|jit_cache)\b", False),
        (r"\bregalloc\b", False),
        (r"\bexecutable\b", False),
        (r"\bCompiledKernel\b", False),
    ]
    return {"E": emit, "P": program}


def scan(src, scope, program_submodules):
    """`[(line, token)]` for every boundary violation in `src`."""
    text = cleaned(src)
    without_reexports = blank_reexports(text)
    hits = set()
    for pattern, exempt_reexports in rules(program_submodules)[scope]:
        haystack = without_reexports if exempt_reexports else text
        for m in re.finditer(pattern, haystack, re.S):
            hits.add((lineno(haystack, m.start()), re.sub(r"\s+", " ", m.group(0))[:70]))
    return sorted(hits)


def program_submodules(program_dir):
    """The submodules of `program/`, read off the filesystem."""
    mods = {p.stem for p in program_dir.glob("*.rs") if p.stem != "mod"}
    return mods | {p.name for p in program_dir.iterdir() if p.is_dir()}


def check_tree():
    subs = program_submodules(SRC / "program")
    failures, files = 0, 0
    for scope, directory in (("E", SRC / "emit"), ("P", SRC / "program")):
        found = sorted(directory.rglob("*.rs"))
        if not found:
            # A renamed or emptied scope would otherwise scan nothing and say OK.
            print(f"FAIL: {directory.relative_to(SRC)}/ holds no .rs files", file=sys.stderr)
            return 1
        for path in found:
            files += 1
            for line, token in scan(path.read_text(), scope, subs):
                failures += 1
                print(
                    f"FAIL: {path.relative_to(SRC)}:{line}: names {token} [rule {scope}]",
                    file=sys.stderr,
                )
    if failures:
        print(f"{failures} emit-boundary violation(s)", file=sys.stderr)
        return 1
    print(f"OK: emit/ and program/ boundaries hold ({files} files)")
    return 0


def self_test():
    subs = {"lower", "scopes", "guards", "layout", "ownership", "tree"}
    cases = []

    def case(name, src, want, scope="E"):
        cases.append((name, len(scan(src, scope, subs)), want))

    token = "use pixelflow_search::egraph::CostModel;"
    case("a plain use", token + "\n", 1)
    case("line comment", "// " + token + "\n", 0)
    case("doc comment", "/// " + token + "\n", 0)
    case("block comment", "/* " + token + " */\n", 0)
    case("string", 'const S: &str = "pixelflow_search::x";\n', 0)
    case("raw string", 'const S: &str = r#"pixelflow_search::x"#;\n', 0)
    case(
        "cfg(test) fn with where commas",
        "#[cfg(test)]\nfn f<T>() where A: X, B: Y {\n  use pixelflow_search::x;\n}\n",
        0,
    )
    case("cfg(test) mod", "#[cfg(test)]\nmod tests {\n  use pixelflow_search::x;\n}\n", 0)
    case("cfg(test) use", "#[cfg(test)]\nuse pixelflow_search::x;\n", 0)
    case(
        "cfg(test) field",
        "struct S {\n  #[cfg(test)]\n  field: pixelflow_search::T,\n  other: u8,\n}\n",
        0,
    )
    case(
        "cfg(test) struct-literal field",
        "fn f() -> S { S {\n  #[cfg(test)]\n  reads: pixelflow_search::Z,\n  b: 1 } }\n",
        0,
    )
    case(
        "cfg(test) statement",
        "fn f() {\n  #[cfg(test)]\n  self.h.set(pixelflow_search::Z);\n}\n",
        0,
    )
    case(
        "cfg(all(test, unix)) mod",
        "#[cfg(all(test, unix))]\nmod m {\n use pixelflow_search::x;\n}\n",
        0,
    )
    case(
        "cfg(not(test)) is production",
        "#[cfg(not(test))]\nfn f() { let _ = pixelflow_search::x; }\n",
        1,
    )
    case(
        "cfg(any(test, unix)) is production",
        "#[cfg(any(test, unix))]\nfn f() { let _ = pixelflow_search::x; }\n",
        1,
    )
    case(
        "violation right after a cfg(test) item with commas in its where-clause",
        "#[cfg(test)]\nfn f<T>() where A: X, B: Y {}\nuse pixelflow_search::x;\n",
        1,
    )
    case(
        "cfg(test) last field with no trailing comma, then a violation",
        "struct S {\n  #[cfg(test)]\n  field: u8\n}\nuse pixelflow_search::x;\n",
        1,
    )
    case(
        "cfg(test) last struct-literal field with no comma, then a violation",
        "fn f() -> S { S { b: 1,\n  #[cfg(test)]\n  reads: 0 } }\nuse pixelflow_search::x;\n",
        1,
    )
    case(
        "multi-line grouped pixelflow_ir use",
        "use pixelflow_ir::{\n kind::OpKind,\n passes::lattice::Collapse,\n};\n",
        1,
    )
    case("pixelflow_ir::kind is fine", "use pixelflow_ir::kind::OpKind;\n", 0)
    case("program vocabulary is fine", "use crate::program::{Def, ScheduledOp};\n", 0)
    case("grouped program submodule", "use crate::program::{Def, scopes::lay_out};\n", 1)
    case("program::tree", "use crate::program::tree::Tree;\n", 1)
    case("program::lower", "use crate::program::lower::arena_to_schedule;\n", 1)
    case("program glob", "use crate::program::*;\n", 1)
    case("pub use of a program submodule is still flagged", "pub use crate::program::tree::Tree;\n", 1)
    case("use of the driver", "use crate::pipeline::compile;\n", 1)
    case("a path into the driver", "fn f() { crate::pipeline::origin(); }\n", 1)
    case("pub use re-export of the driver is exempt", "pub use crate::pipeline::{compile, origin};\n", 0)
    case(
        "multi-line pub use re-export of the driver is exempt",
        "pub use crate::pipeline::{\n    compile,\n    origin,\n};\n",
        0,
    )
    case("pub(crate) use of the driver is not a re-export", "pub(crate) use crate::pipeline::compile;\n", 1)
    case("the exemption ends at the statement", "pub use crate::pipeline::compile;\nuse crate::pipeline::origin;\n", 1)
    case("pub use of the cache is still flagged", "pub use crate::jit_cache::compile;\n", 1)
    case("pub use of the search is still flagged", "pub use pixelflow_search::egraph::CostModel;\n", 1)
    case("grouped crate:: use of the driver", "use crate::{error, pipeline::compile};\n", 1)
    case("grouped crate:: use of the cache", "use crate::{jit_cache::compile};\n", 1)
    case("multi-line grouped crate:: use", "use crate::{\n    error,\n    jit_cache::K,\n};\n", 1)
    case("grouped super::super:: use of the driver", "use super::super::{pipeline};\n", 1)
    case("grouped crate:: use of other things is fine", "use crate::{error::CompileError, program::Def};\n", 0)
    case("a name that only starts with pipeline", "use crate::{pipeline_stats::X};\n", 0)
    case("pub use of a grouped driver path is exempt", "pub use crate::{pipeline::{compile, origin}};\n", 0)
    case("P: grouped crate:: use of the emitter", "use crate::{emit::x};\n", 1, "P")
    case("P: grouped crate:: use of the isa", "use crate::{error, isa::Isa};\n", 1, "P")
    case("P: grouped crate:: use of other things is fine", "use crate::{error::CompileError};\n", 0, "P")
    case(
        "cfg(test) if statement, then a violation",
        "fn f() {\n  #[cfg(test)]\n  if c { x(); }\n  crate::pipeline::compile();\n}\n",
        1,
    )
    case(
        "cfg(test) match statement, then a violation",
        "fn f() {\n  #[cfg(test)]\n  match c { _ => {} }\n  crate::pipeline::compile();\n}\n",
        1,
    )
    case(
        "cfg(test) unsafe block, then a violation",
        "fn f() {\n  #[cfg(test)]\n  unsafe { g(); }\n  crate::pipeline::compile();\n}\n",
        1,
    )
    case(
        "cfg(test) if statement hides what is inside it",
        "fn f() {\n  #[cfg(test)]\n  if c { crate::pipeline::compile(); }\n}\n",
        0,
    )
    case("P: emit", "use crate::emit::guards::X;\n", 1, "P")
    case("P: regalloc", "fn f(d: regalloc::Def) {}\n", 1, "P")
    case("P: executable and CompiledKernel", "use executable::CompiledKernel;\n", 2, "P")
    case("P: pipeline", "use crate::pipeline::compile;\n", 1, "P")
    case("P: pub use does not exempt the pipeline", "pub use crate::pipeline::compile;\n", 1, "P")
    case("P may name the search", token + "\n", 0, "P")
    case("P: a cfg(test) item may name the driver", "#[cfg(test)]\nmod tests {\n use crate::pipeline::schedule_for;\n}\n", 0, "P")
    case(
        "a lifetime is not a char literal",
        "fn f<'a>(x: &'a str) -> &'a str { x }\nuse pixelflow_search::x;\n",
        1,
    )
    case("a quote char literal", "const Q: char = '\"';\nuse pixelflow_search::x;\n", 1)

    failed = [c for c in cases if c[1] != c[2]]
    for name, got, want in failed:
        print(f"self-test FAIL: {name}: got {got}, want {want}", file=sys.stderr)
    if failed:
        return 1
    print(f"self-test OK: {len(cases)} cases")
    return 0


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    if sys.argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    return check_tree()


if __name__ == "__main__":
    sys.exit(main())
